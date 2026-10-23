"""Stores of the model's extraction of a query, so a repeated query reads the same way.

`extract()` consults the store before it calls the model and writes the model's
`Extraction` JSON to it after a valid parse, before any post-processing. A key
(`extraction_key`) hashes the query as sent, the extract version (the system
prompt and the `Extraction` schema) and the model id, seed and thinking level, so
any edit to the prompt or schema reads afresh. `SEARCH_EXTRACT_STORE` selects the
store: `off`, `memory` (an in-process LRU bounded in bytes, with no TTL) or
`file:<path>` (JSON lines, recorded by one run and replayed by others).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# The in-process store's byte budget. An extraction is under 1 KiB of JSON, so it
# holds over 16,000 queries.
MEMORY_STORE_MAX_BYTES = 16 * 1024 * 1024


def extraction_key(query: str, *, version: str, year: int, identity: Mapping[str, object]) -> str:
    """Hash `query` as sent to the model, the extract `version`, the `year` the prompt
    names and the model's `identity` (its id, seed and thinking level).

    The exact text keys the reading because `extract()` checks the reading against it:
    role spans, operation phrases and restored word order all read the query's own text.
    """
    document = {
        "query": query,
        "version": version,
        "year": year,
        "identity": identity,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "extract_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExtractionStore(Protocol):
    """The model's `Extraction` JSON by `extraction_key`."""

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, value: str) -> None: ...


class MemoryExtractionStore:
    """An LRU of extractions bounded in bytes, held for the life of the process.

    An entry weighs its key and value in UTF-8. A value larger than the whole budget
    is not kept.
    """

    def __init__(self, *, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._bytes = 0
        self._entries: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: str) -> None:
        size = _weight(key, value)
        if size > self._max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= _weight(key, previous)
        self._entries[key] = value
        self._bytes += size
        while self._bytes > self._max_bytes:
            evicted_key, evicted = self._entries.popitem(last=False)
            self._bytes -= _weight(evicted_key, evicted)


def _weight(key: str, value: str) -> int:
    return len(key.encode("utf-8")) + len(value.encode("utf-8"))


class FileExtractionStore:
    """Extractions recorded as JSON lines, one `{"key", "value"}` object per line.

    Each `put` appends its line in one unbuffered write to a file opened for append,
    so writers in several processes do not interleave lines. A `get` that misses
    reads the lines appended since the last read, so a second process sees what the
    first recorded. A trailing line without its newline is still being written and is
    read once it is complete. A complete line that is not a `{"key", "value"}` object is
    logged and skipped. The last line for a key wins.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, str] = {}
        self._offset = 0
        self._read_appended()

    def get(self, key: str) -> str | None:
        if key not in self._entries:
            self._read_appended()
        return self._entries.get(key)

    def put(self, key: str, value: str) -> None:
        line = json.dumps({"key": key, "value": value}, ensure_ascii=True) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab", buffering=0) as file:
            file.write(line.encode("utf-8"))
        self._entries[key] = value

    def _read_appended(self) -> None:
        try:
            with self._path.open("rb") as file:
                file.seek(self._offset)
                appended = file.read()
        except FileNotFoundError:
            return
        complete = appended[: appended.rfind(b"\n") + 1]
        self._offset += len(complete)
        for line in complete.splitlines():
            try:
                record = json.loads(line)
                self._entries[record["key"]] = record["value"]
            except (ValueError, TypeError, KeyError):
                logger.warning("Extraction store %s: skipping a malformed line", self._path)


def is_extraction_store_mode(mode: str) -> bool:
    """Whether `mode` is a `SEARCH_EXTRACT_STORE` value: off, memory or file:<path>."""
    return mode in {"off", "memory"} or (mode.startswith("file:") and len(mode) > len("file:"))


def build_extraction_store(mode: str) -> ExtractionStore | None:
    """Build the store `SEARCH_EXTRACT_STORE` names, or `None` for `off`."""
    if not is_extraction_store_mode(mode):
        raise ValueError(f"SEARCH_EXTRACT_STORE must be off, memory or file:<path>; got {mode!r}")
    if mode == "off":
        return None
    if mode == "memory":
        return MemoryExtractionStore(max_bytes=MEMORY_STORE_MAX_BYTES)
    return FileExtractionStore(Path(mode.removeprefix("file:")))
