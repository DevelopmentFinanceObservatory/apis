"""Pins `file://` generation-checked reads, writes, and recursive listings."""

from __future__ import annotations

import os
import time

from artifact_publication.uri import (
    list_uri_object_times,
    read_uri_bytes_and_generation,
    write_uri_bytes,
    write_uri_bytes_if_unchanged,
)


def _generation(uri: str) -> str | None:
    found = read_uri_bytes_and_generation(uri)
    return None if found is None else found[1]


def test_read_uri_bytes_and_generation_is_none_for_a_missing_object(tmp_path):
    assert read_uri_bytes_and_generation(f"file://{tmp_path / 'nothing.json'}") is None


def test_read_uri_bytes_and_generation_returns_the_bytes_it_tokenised(tmp_path):
    uri = f"file://{tmp_path / 'current.json'}"
    write_uri_bytes(uri, b"one")
    found = read_uri_bytes_and_generation(uri)
    assert found is not None
    data, generation = found
    assert data == b"one"

    write_uri_bytes(uri, b"one")  # identical content -> identical generation
    assert _generation(uri) == generation

    write_uri_bytes(uri, b"two")
    assert _generation(uri) != generation


def test_write_uri_bytes_if_unchanged_requires_absence_when_expected_generation_is_none(
    tmp_path,
):
    uri = f"file://{tmp_path / 'current.json'}"
    assert write_uri_bytes_if_unchanged(uri, b"first", expected_generation=None) is True
    assert (tmp_path / "current.json").read_bytes() == b"first"

    # A second write also claiming "nothing there yet" loses: the object
    # now exists.
    assert write_uri_bytes_if_unchanged(uri, b"second", expected_generation=None) is False
    assert (tmp_path / "current.json").read_bytes() == b"first"


def test_write_uri_bytes_if_unchanged_refuses_on_a_stale_generation(tmp_path):
    uri = f"file://{tmp_path / 'current.json'}"
    write_uri_bytes(uri, b"first")
    stale_generation = _generation(uri)

    write_uri_bytes(uri, b"concurrent")  # someone else publishes in between

    assert write_uri_bytes_if_unchanged(uri, b"mine", expected_generation=stale_generation) is False
    assert (tmp_path / "current.json").read_bytes() == b"concurrent"


def test_write_uri_bytes_if_unchanged_succeeds_when_the_generation_still_matches(tmp_path):
    uri = f"file://{tmp_path / 'current.json'}"
    write_uri_bytes(uri, b"first")
    current_generation = _generation(uri)

    published = write_uri_bytes_if_unchanged(uri, b"second", expected_generation=current_generation)

    assert published is True
    assert (tmp_path / "current.json").read_bytes() == b"second"


def test_list_uri_object_times_reads_every_file_recursively(tmp_path):
    base = tmp_path / "format-1"
    (base / "versions" / "b1").mkdir(parents=True)
    (base / "current.json").write_text("{}")
    old_file = base / "versions" / "b1" / "a.txt"
    old_file.write_text("x")
    old_time = time.time() - 1000
    os.utime(old_file, (old_time, old_time))
    (base / "versions" / "b1" / "b.txt").write_text("y")

    times = list_uri_object_times(f"file://{base}")

    assert set(times) == {"current.json", "versions/b1/a.txt", "versions/b1/b.txt"}
    assert abs(times["versions/b1/a.txt"] - old_time) < 2
    assert times["versions/b1/b.txt"] > old_time + 900


def test_list_uri_object_times_is_empty_for_a_base_that_does_not_exist(tmp_path):
    assert list_uri_object_times(f"file://{tmp_path / 'missing'}") == {}
