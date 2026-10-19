"""Reads and writes artifact bytes for `gs://` and `file://` URIs.

GCS libraries load lazily. `gs://` deployments need `dfo-data-tools[coverage]`.
Missing paths raise `FileNotFoundError`. Failed conditional writes return
`False`. Listing a missing prefix returns an empty map, and deleting one is a
no-op. Serving code reads artifacts. `series_coverage.job` and
`origin_catalogue.job` write and prune them.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlsplit


class ArtifactUriError(ValueError):
    """Raised for unsupported URI schemes or GCS URIs with no bucket."""


def join_uri(base_uri: str, *parts: str) -> str:
    """Appends path segments with one `/`, preserving a `gs://` scheme."""
    joined = base_uri.rstrip("/")
    for part in parts:
        joined = f"{joined}/{part.strip('/')}"
    return joined


def read_uri_bytes(uri: str) -> bytes:
    """Reads the object in full. Invalid schemes or an empty GCS bucket raise
    `ArtifactUriError`. Missing files or objects raise `FileNotFoundError`.
    """
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    if parsed.scheme == "gs":
        return _read_gs(bucket=parsed.netloc, object_name=parsed.path.lstrip("/"))
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _read_gs(*, bucket: str, object_name: str) -> bytes:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (object {object_name!r})")
    from google.api_core.exceptions import NotFound
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(object_name)
    try:
        return blob.download_as_bytes()
    except NotFound as exc:
        raise FileNotFoundError(f"gs://{bucket}/{object_name} does not exist") from exc


def write_uri_bytes(uri: str, data: bytes) -> None:
    """Writes all bytes, creating missing parent directories for `file://`.

    Invalid schemes or an empty GCS bucket raise `ArtifactUriError`.
    """
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    if parsed.scheme == "gs":
        _write_gs(bucket=parsed.netloc, object_name=parsed.path.lstrip("/"), data=data)
        return
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _write_gs(*, bucket: str, object_name: str, data: bytes) -> None:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (object {object_name!r})")
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(object_name)
    blob.upload_from_string(data)


def download_uri_to_path(uri: str, path: Path) -> None:
    """Streams the object to `path`, overwriting it, without holding it in
    memory. Missing files or objects raise `FileNotFoundError`, and invalid
    schemes or an empty GCS bucket raise `ArtifactUriError`."""
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        shutil.copyfile(parsed.path, path)
        return
    if parsed.scheme == "gs":
        _download_gs(bucket=parsed.netloc, object_name=parsed.path.lstrip("/"), path=path)
        return
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _download_gs(*, bucket: str, object_name: str, path: Path) -> None:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (object {object_name!r})")
    from google.api_core.exceptions import NotFound
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(object_name)
    try:
        blob.download_to_filename(str(path))
    except NotFound as exc:
        raise FileNotFoundError(f"gs://{bucket}/{object_name} does not exist") from exc


def read_uri_bytes_and_generation(uri: str) -> tuple[bytes, str] | None:
    """Returns bytes and a version token from one read, or `None` if absent.

    GCS uses the object's generation. Local files use SHA-256. Other read
    failures propagate, and unsupported schemes raise `ArtifactUriError`.
    """
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        return data, hashlib.sha256(data).hexdigest()
    if parsed.scheme == "gs":
        return _read_gs_with_generation(bucket=parsed.netloc, object_name=parsed.path.lstrip("/"))
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _read_file_generation(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_gs_with_generation(*, bucket: str, object_name: str) -> tuple[bytes, str] | None:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (object {object_name!r})")
    from google.api_core.exceptions import NotFound
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(object_name)
    try:
        data = blob.download_as_bytes()
    except NotFound:
        return None
    if blob.generation is None:
        raise RuntimeError(f"gs://{bucket}/{object_name} downloaded with no generation header")
    return data, str(blob.generation)


def write_uri_bytes_if_unchanged(uri: str, data: bytes, *, expected_generation: str | None) -> bool:
    """Writes only when the current token matches `expected_generation`.
    `None` requires absence. Returns whether the write landed.

    Build files use create-only writes. `current.json` uses a generation
    token. GCS checks atomically. Identical bytes after a precondition failure
    mean this write committed but its response was lost. Local files recheck
    the token before writing and use one write attempt.
    """
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        return _write_file_if_unchanged(
            Path(parsed.path), data, expected_generation=expected_generation
        )
    if parsed.scheme == "gs":
        return _write_gs_if_unchanged(
            bucket=parsed.netloc,
            object_name=parsed.path.lstrip("/"),
            data=data,
            expected_generation=expected_generation,
        )
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _write_file_if_unchanged(path: Path, data: bytes, *, expected_generation: str | None) -> bool:
    # The generation check and local write run once, so no retry can obscure
    # whether this write landed.
    if _read_file_generation(path) != expected_generation:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def _write_gs_if_unchanged(
    *, bucket: str, object_name: str, data: bytes, expected_generation: str | None
) -> bool:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (object {object_name!r})")
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(object_name)
    match = int(expected_generation) if expected_generation is not None else 0
    try:
        blob.upload_from_string(data, if_generation_match=match)
    except PreconditionFailed:
        # GCS retries uploads by default. If the first conditional write commits
        # but its response is lost, the retry fails against the identical bytes.
        # Matching bytes confirm this write. Missing or different bytes return False.
        try:
            current = _read_gs(bucket=bucket, object_name=object_name)
        except FileNotFoundError:
            return False
        return current == data
    return True


def list_uri_object_times(uri: str) -> dict[str, float]:
    """Maps each object under `uri` to its relative path and write time.

    GCS uses `time_created` from one recursive list. Local files use `st_mtime`.
    A missing prefix returns an empty mapping.
    """
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        return _list_file_object_times(Path(parsed.path))
    if parsed.scheme == "gs":
        return _list_gs_object_times(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _list_file_object_times(path: Path) -> dict[str, float]:
    if not path.is_dir():
        return {}
    return {
        found.relative_to(path).as_posix(): found.stat().st_mtime
        for found in path.rglob("*")
        if found.is_file()
    }


def _list_gs_object_times(*, bucket: str, prefix: str) -> dict[str, float]:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (prefix {prefix!r})")
    from google.cloud import storage

    client = storage.Client()
    prefix = prefix.rstrip("/") + "/"
    return {
        blob.name[len(prefix) :]: blob.time_created.timestamp()
        for blob in client.list_blobs(bucket, prefix=prefix)
        if blob.time_created is not None
    }


def list_uri_object_generations(uri: str) -> dict[str, tuple[float, str]]:
    """Maps each object under `uri` to its relative path, write time and the
    version token `delete_uri_if_unchanged` takes, without downloading it.

    GCS reads both from one recursive list. Local files use `st_mtime` and
    stream each file through SHA-256. A missing prefix returns an empty
    mapping.
    """
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        return _list_file_object_generations(Path(parsed.path))
    if parsed.scheme == "gs":
        return _list_gs_object_generations(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _list_file_object_generations(path: Path) -> dict[str, tuple[float, str]]:
    if not path.is_dir():
        return {}
    listed = {}
    for found in path.rglob("*"):
        if found.is_file():
            with found.open("rb") as handle:
                token = hashlib.file_digest(handle, "sha256").hexdigest()
            listed[found.relative_to(path).as_posix()] = (found.stat().st_mtime, token)
    return listed


def _list_gs_object_generations(*, bucket: str, prefix: str) -> dict[str, tuple[float, str]]:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (prefix {prefix!r})")
    from google.cloud import storage

    client = storage.Client()
    prefix = prefix.rstrip("/") + "/"
    return {
        blob.name[len(prefix) :]: (blob.time_created.timestamp(), str(blob.generation))
        for blob in client.list_blobs(bucket, prefix=prefix)
        if blob.time_created is not None and blob.generation is not None
    }


def delete_uri_if_unchanged(uri: str, *, expected_generation: str) -> bool:
    """Deletes one object only while its token equals `expected_generation`.
    Returns whether this call deleted it, so an absent or replaced object
    returns `False`. GCS checks atomically. Local files recheck the token
    before unlinking."""
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        if _read_file_generation(path) != expected_generation:
            return False
        path.unlink(missing_ok=True)
        return True
    if parsed.scheme == "gs":
        return _delete_gs_if_unchanged(
            bucket=parsed.netloc,
            object_name=parsed.path.lstrip("/"),
            expected_generation=expected_generation,
        )
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _delete_gs_if_unchanged(*, bucket: str, object_name: str, expected_generation: str) -> bool:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (object {object_name!r})")
    from google.api_core.exceptions import NotFound, PreconditionFailed
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(object_name)
    try:
        blob.delete(if_generation_match=int(expected_generation))
    except (NotFound, PreconditionFailed):
        return False
    return True


def delete_uri_prefix(uri: str) -> None:
    """Deletes a `versions/<build_id>/` tree for job pruning.

    GCS lists and deletes each blob. Local paths remove the directory tree.
    Missing prefixes are no-ops.
    """
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        if path.exists():
            shutil.rmtree(path)
        return
    if parsed.scheme == "gs":
        _delete_gs_prefix(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
        return
    raise ArtifactUriError(
        f"unsupported scheme {parsed.scheme!r} in URI {uri!r}; expected gs:// or file://"
    )


def _delete_gs_prefix(*, bucket: str, prefix: str) -> None:
    if not bucket:
        raise ArtifactUriError(f"gs:// URI names no bucket (prefix {prefix!r})")
    from google.api_core.exceptions import NotFound
    from google.cloud import storage

    client = storage.Client()
    prefix = prefix.rstrip("/") + "/"
    for blob in client.list_blobs(bucket, prefix=prefix):
        # A concurrent prune of the same build can delete this blob between
        # this listing and this delete -- the docstring's own "two prune
        # runs racing the same build is not an error" guarantee, so a
        # blob already gone by the time we get here is a no-op, not a
        # propagated `NotFound`.
        with contextlib.suppress(NotFound):
            blob.delete()


__all__ = [
    "ArtifactUriError",
    "delete_uri_if_unchanged",
    "delete_uri_prefix",
    "download_uri_to_path",
    "join_uri",
    "list_uri_object_generations",
    "list_uri_object_times",
    "read_uri_bytes",
    "read_uri_bytes_and_generation",
    "write_uri_bytes",
    "write_uri_bytes_if_unchanged",
]
