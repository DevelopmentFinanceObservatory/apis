"""Durable, resumable storage for captures and cells.

Each store has a manifest, append-only JSONL logs, and status. Cells also have `scores.json`.
The 21-cell ablation sweep rebuilds its resume index to preserve each record once.

Each append flushes its JSON line. `close()` fsyncs the log. Reopening rebuilds the key index,
repairs a malformed trailing line, and raises on earlier corruption or duplicate keys. Opening a
capture keeps binds whose case has a durable extract, so interrupted cases are rerun.

`finish()` closes and fsyncs logs, writes scores, then writes `status.json` last. Its presence
proves the records and scores it describes are durable. `is_complete()` and `require_complete()`
gate use of completed stores.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CallKey = tuple[str, int, int]  # (case_id, leg, seed)
BindKey = tuple[str, int]  # (case_id, leg)


class StoreError(ValueError):
    """A store operation violated a durability or resume invariant."""


class NotComplete(StoreError):
    """Raised by `require_complete()` naming the store that failed it."""


def _replay[K](path: Path, *, key_fn: Callable[[dict[str, Any]], K]) -> list[dict[str, Any]]:
    """Replay complete JSONL records, dropping and fsyncing a malformed trailing line.

    Other malformed lines and duplicate keys raise `StoreError`.
    """
    lines = path.read_text().splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    seen: set[K] = set()
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise StoreError(f"{path}: malformed record on line {i + 1}") from None
            good = "".join(lines[:i])
            with path.open("w") as f:
                f.write(good)
                f.flush()
                os.fsync(f.fileno())
            break
        key = key_fn(record)
        if key in seen:
            raise StoreError(f"{path}: duplicate key {key!r} on line {i + 1}")
        seen.add(key)
        records.append(record)
    return records


class _AppendLog[K]:
    """A keyed JSONL log for append and resume. Duplicate keys raise `StoreError`.

    A lock guards reads and writes. The driver semaphore bounds concurrent calls.
    """

    def __init__(self, path: Path, *, key_fn: Callable[[dict[str, Any]], K]) -> None:
        self._path = path
        self._key_fn = key_fn
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = _replay(path, key_fn=key_fn) if path.exists() else []
        self._keys: set[K] = {key_fn(r) for r in self._records}
        self._fh = path.open("a")

    def keys(self) -> set[K]:
        with self._lock:
            return set(self._keys)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def append(self, record: dict[str, Any]) -> None:
        key = self._key_fn(record)
        with self._lock:
            if key in self._keys:
                raise StoreError(f"{self._path}: key already written: {key!r}")
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
            self._keys.add(key)
            self._records.append(record)

    def close(self) -> None:
        """Flush, fsync, and close the log.

        Safe to call again during cleanup.
        """
        with self._lock:
            if self._fh.closed:
                return
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def drop(self, keys_to_drop: set[K]) -> None:
        """Rewrite and fsync the log without the selected keys.

        Capture-open repair calls this before appends. An empty key set leaves the file untouched.
        """
        with self._lock:
            if not keys_to_drop:
                return
            kept = [r for r in self._records if self._key_fn(r) not in keys_to_drop]
            self._fh.close()
            with self._path.open("w") as f:
                for record in kept:
                    f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._records = kept
            self._keys = {self._key_fn(r) for r in kept}
            self._fh = self._path.open("a")


def _write_json_durable(path: Path, data: Any) -> None:
    with path.open("w") as f:
        f.write(json.dumps(data, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class CaptureManifest:
    """Capture identity and replay settings.

    Replay uses the captured `substituted_date` so bind menus match extraction prompts.
    """

    capture_id: str
    git_sha: str
    engine_build_id: str
    model: str
    top_k: int
    substituted_date: str
    graph_base: str
    corpus_sha256: str
    seeds: tuple[int, ...]
    extract_system_sha1: str
    bind_system_sha1: str


@dataclass(frozen=True)
class ExtractRecord:
    """One extract call per case, replayed by every downstream cell and seed."""

    case_id: str
    query: str
    system_sha1: str
    output: dict[str, Any]
    usage: dict[str, Any] | None
    elapsed_ms: int


@dataclass(frozen=True)
class BindRecord:
    """One bind call per `(case_id, leg)`, captured at the run's top-k.

    Smaller-k cells truncate the stored `prompt`.
    """

    case_id: str
    leg: int
    variable: str
    prompt: str
    prompt_sha1: str
    system_sha1: str
    output: dict[str, Any]
    usage: dict[str, Any] | None
    elapsed_ms: int


def _dump_extract(record: ExtractRecord) -> dict[str, Any]:
    return {
        "case_id": record.case_id,
        "query": record.query,
        "system_sha1": record.system_sha1,
        "output": record.output,
        "usage": record.usage,
        "elapsed_ms": record.elapsed_ms,
    }


def _load_extract(data: dict[str, Any]) -> ExtractRecord:
    return ExtractRecord(**data)


def _dump_bind(record: BindRecord) -> dict[str, Any]:
    return {
        "case_id": record.case_id,
        "leg": record.leg,
        "variable": record.variable,
        "prompt": record.prompt,
        "prompt_sha1": record.prompt_sha1,
        "system_sha1": record.system_sha1,
        "output": record.output,
        "usage": record.usage,
        "elapsed_ms": record.elapsed_ms,
    }


def _load_bind(data: dict[str, Any]) -> BindRecord:
    return BindRecord(**data)


def _dump_capture_manifest(manifest: CaptureManifest) -> dict[str, Any]:
    return {
        "capture_id": manifest.capture_id,
        "git_sha": manifest.git_sha,
        "engine_build_id": manifest.engine_build_id,
        "model": manifest.model,
        "top_k": manifest.top_k,
        "substituted_date": manifest.substituted_date,
        "graph_base": manifest.graph_base,
        "corpus_sha256": manifest.corpus_sha256,
        "seeds": list(manifest.seeds),
        "extract_system_sha1": manifest.extract_system_sha1,
        "bind_system_sha1": manifest.bind_system_sha1,
    }


def _load_capture_manifest(data: dict[str, Any]) -> CaptureManifest:
    return CaptureManifest(
        capture_id=data["capture_id"],
        git_sha=data["git_sha"],
        engine_build_id=data["engine_build_id"],
        model=data["model"],
        top_k=data["top_k"],
        substituted_date=data["substituted_date"],
        graph_base=data["graph_base"],
        corpus_sha256=data["corpus_sha256"],
        seeds=tuple(data["seeds"]),
        extract_system_sha1=data["extract_system_sha1"],
        bind_system_sha1=data["bind_system_sha1"],
    )


class CaptureStore:
    """Capture files: manifest, extract/bind logs, graph database path, and status.

    Use `create` for a fresh capture or `open` to resume. Both check the manifest.
    """

    _MANIFEST = "manifest.json"
    _EXTRACT = "extract.jsonl"
    _BIND = "bind.jsonl"
    _GRAPH_DB = "graph.sqlite"
    _STATUS = "status.json"

    def __init__(self, *, root: Path, manifest: CaptureManifest) -> None:
        self.root = root
        self.manifest = manifest
        self._extract_log: _AppendLog[str] = _AppendLog(
            root / self._EXTRACT, key_fn=lambda r: r["case_id"]
        )
        self._bind_log: _AppendLog[BindKey] = _AppendLog(
            root / self._BIND, key_fn=lambda r: (r["case_id"], r["leg"])
        )
        self._drop_orphaned_binds()
        self._finished = False

    def _drop_orphaned_binds(self) -> None:
        """Remove orphaned binds on open so `run_capture` can rerun their case.

        The fresh extraction may change its variable split, so the stale bind record must be
        removed from the append-only log before a new append. Finished captures are unchanged.
        """
        if (self.root / self._STATUS).is_file():
            return
        done_extract = self._extract_log.keys()
        orphaned = {key for key in self._bind_log.keys() if key[0] not in done_extract}  # noqa: SIM118 -- _AppendLog exposes `.keys()` only, no `__iter__`
        self._bind_log.drop(orphaned)

    @classmethod
    def create(cls, root: Path, *, manifest: CaptureManifest) -> CaptureStore:
        """Creates a fresh capture store at `root`. Raises `StoreError` if
        a manifest already exists there -- resume with `open` instead."""
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / cls._MANIFEST
        if manifest_path.is_file():
            raise StoreError(f"{root}: capture store already exists (found {cls._MANIFEST})")
        _write_json_durable(manifest_path, _dump_capture_manifest(manifest))
        return cls(root=root, manifest=manifest)

    @classmethod
    def open(cls, root: Path) -> CaptureStore:
        """Reopens an existing capture store at `root` for resumption or
        for reading. Raises `StoreError` if `root` has no manifest."""
        manifest_path = root / cls._MANIFEST
        if not manifest_path.is_file():
            raise StoreError(f"{root}: no {cls._MANIFEST}; nothing to resume")
        manifest = _load_capture_manifest(_read_json(manifest_path))
        return cls(root=root, manifest=manifest)

    def __enter__(self) -> CaptureStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def graph_db_path(self) -> Path:
        """Path for `GraphCache` to open for this capture."""
        return self.root / self._GRAPH_DB

    def done_extract_keys(self) -> set[str]:
        return self._extract_log.keys()

    def done_bind_keys(self) -> set[BindKey]:
        return self._bind_log.keys()

    def extract_records(self) -> tuple[ExtractRecord, ...]:
        return tuple(_load_extract(r) for r in self._extract_log.records())

    def bind_records(self) -> tuple[BindRecord, ...]:
        return tuple(_load_bind(r) for r in self._bind_log.records())

    def append_extract(self, record: ExtractRecord) -> None:
        if self._finished:
            raise StoreError(f"{self.root}: capture store is finished; no further appends")
        self._extract_log.append(_dump_extract(record))

    def append_bind(self, record: BindRecord) -> None:
        if self._finished:
            raise StoreError(f"{self.root}: capture store is finished; no further appends")
        self._bind_log.append(_dump_bind(record))

    def status(self) -> dict[str, Any]:
        """Return status and sorted `done_extract`/`done_bind` keys.

        Read the on-disk status after finish. Partial stores use the resume indexes.
        """
        status_path = self.root / self._STATUS
        if status_path.is_file():
            return _read_json(status_path)
        return {
            "state": "partial",
            "done_extract": sorted(self.done_extract_keys()),
            "done_bind": sorted(self.done_bind_keys()),
        }

    def is_complete(self) -> bool:
        return self.status()["state"] == "complete"

    def require_complete(self) -> None:
        """Raises `NotComplete` unless this capture is finished. A rung
        naming a capture must call this before replaying it."""
        if not self.is_complete():
            raise NotComplete(f"{self.root}: capture is not complete")

    def finish(self) -> None:
        """Close and fsync both logs, then write status last to mark their records durable."""
        if self._finished:
            raise StoreError(f"{self.root}: capture store already finished")
        self._extract_log.close()
        self._bind_log.close()
        _write_json_durable(
            self.root / self._STATUS,
            {
                "state": "complete",
                "done_extract": sorted(self.done_extract_keys()),
                "done_bind": sorted(self.done_bind_keys()),
            },
        )
        self._finished = True

    def close(self) -> None:
        """Closes both logs without marking the store complete -- for a
        driver run stopping short (an error, an interrupt, or simply
        deferring `finish()` to a later `--resume`)."""
        if self._finished:
            return
        self._extract_log.close()
        self._bind_log.close()


@dataclass(frozen=True)
class CellManifest:
    """Cell identity and replay inputs.

    `rung` stores JSON-safe resolved data, so reports stay readable when block files change.
    System fields hold the composed prompts verbatim.
    """

    cell_id: str
    rung: dict[str, Any]
    capture_id: str
    git_sha: str
    model: str
    seeds: tuple[int, ...]
    extract_system: str
    bind_system: str


def _dump_cell_manifest(manifest: CellManifest) -> dict[str, Any]:
    return {
        "cell_id": manifest.cell_id,
        "rung": manifest.rung,
        "capture_id": manifest.capture_id,
        "git_sha": manifest.git_sha,
        "model": manifest.model,
        "seeds": list(manifest.seeds),
        "extract_system": manifest.extract_system,
        "bind_system": manifest.bind_system,
    }


def _load_cell_manifest(data: dict[str, Any]) -> CellManifest:
    return CellManifest(
        cell_id=data["cell_id"],
        rung=data["rung"],
        capture_id=data["capture_id"],
        git_sha=data["git_sha"],
        model=data["model"],
        seeds=tuple(data["seeds"]),
        extract_system=data["extract_system"],
        bind_system=data["bind_system"],
    )


class CellStore:
    """Cell files: manifest, `calls.jsonl`, scores, and status.

    `append` flattens each call mapping with its `(case_id, leg, seed)` key, without importing
    the caller's `CallRecord` type. Use `create` or `open` to initialise the store.
    """

    _MANIFEST = "manifest.json"
    _CALLS = "calls.jsonl"
    _SCORES = "scores.json"
    _STATUS = "status.json"

    def __init__(self, *, root: Path, manifest: CellManifest) -> None:
        self.root = root
        self.manifest = manifest
        self._log: _AppendLog[CallKey] = _AppendLog(
            root / self._CALLS,
            key_fn=lambda r: (r["case_id"], r["leg"], r["seed"]),
        )
        self._finished = False

    @classmethod
    def create(cls, root: Path, *, manifest: CellManifest) -> CellStore:
        """Creates a fresh cell store at `root`. Raises `StoreError` if a
        manifest already exists there -- resume with `open` instead."""
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / cls._MANIFEST
        if manifest_path.is_file():
            raise StoreError(f"{root}: cell store already exists (found {cls._MANIFEST})")
        _write_json_durable(manifest_path, _dump_cell_manifest(manifest))
        return cls(root=root, manifest=manifest)

    @classmethod
    def open(cls, root: Path) -> CellStore:
        """Reopens an existing cell store at `root` for resumption or for
        reading. Raises `StoreError` if `root` has no manifest."""
        manifest_path = root / cls._MANIFEST
        if not manifest_path.is_file():
            raise StoreError(f"{root}: no {cls._MANIFEST}; nothing to resume")
        manifest = _load_cell_manifest(_read_json(manifest_path))
        return cls(root=root, manifest=manifest)

    def __enter__(self) -> CellStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def done_keys(self) -> set[CallKey]:
        return self._log.keys()

    def call_records(self) -> tuple[dict[str, Any], ...]:
        """Return flattened call records for reporting and offline rescoring."""
        return tuple(self._log.records())

    def append(self, *, case_id: str, leg: int, seed: int, call: Mapping[str, Any]) -> None:
        if self._finished:
            raise StoreError(f"{self.root}: cell store is finished; no further appends")
        record = {"case_id": case_id, "leg": leg, "seed": seed, **call}
        self._log.append(record)

    def status(self) -> dict[str, Any]:
        """Return status and sorted `done` keys from disk or the resume index."""
        status_path = self.root / self._STATUS
        if status_path.is_file():
            return _read_json(status_path)
        return {"state": "partial", "done": [list(k) for k in sorted(self.done_keys())]}

    def is_complete(self) -> bool:
        return self.status()["state"] == "complete"

    def require_complete(self) -> None:
        """Raise `NotComplete` until the cell is finished.

        Reports call this before reading scores unless `--allow-partial` is set.
        """
        if not self.is_complete():
            raise NotComplete(f"{self.root}: cell is not complete")

    def read_scores(self) -> dict[str, Any]:
        scores_path = self.root / self._SCORES
        if not scores_path.is_file():
            raise StoreError(f"{self.root}: no {self._SCORES} yet; call finish() first")
        return _read_json(scores_path)

    def finish(self, scores: Mapping[str, Any]) -> None:
        """Close and fsync calls, write scores, then write status last to mark both durable."""
        if self._finished:
            raise StoreError(f"{self.root}: cell store already finished")
        self._log.close()
        _write_json_durable(self.root / self._SCORES, dict(scores))
        _write_json_durable(
            self.root / self._STATUS,
            {"state": "complete", "done": [list(k) for k in sorted(self.done_keys())]},
        )
        self._finished = True

    def close(self) -> None:
        """Closes the log without marking the store complete -- for a
        driver run stopping short."""
        if self._finished:
            return
        self._log.close()
