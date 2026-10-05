"""Checks Dockerfile cache paths and warm-script postconditions for ResolveKit."""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WARM_SCRIPT = _REPO_ROOT / "scripts" / "warm_resolvekit_cache.py"


def _warm_module():
    """Load the warm script as a module so its helpers can be tested directly.

    `scripts/` is not a package, so this goes through the file path. Importing
    runs no work of its own: everything the script does lives under `main()`,
    behind the `__main__` guard. That matters here, because `main()` rewrites
    mtimes inside the installed resolvekit package, which is a side effect the
    test suite must not have on a developer's own environment.
    """
    spec = importlib.util.spec_from_file_location("warm_resolvekit_cache", _WARM_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ENV_CACHE_DIR_RE = re.compile(r"RESOLVEKIT_CACHE_DIR=(\S+)")
_WARM_RUN_RE = re.compile(r"RUN\s+\S*python\S*\s+scripts/warm_resolvekit_cache\.py")
# The build stage's second `uv sync` -- the one that installs the project
# itself, after `COPY . ./` -- is the only `RUN uv sync` with `--no-editable`.
# A `RUN` step's `--mount=...` flags sit between `RUN` and the command on
# their own continuation lines, so this allows any number of them ahead of
# `uv sync`, and lets the command itself span lines up to `--no-editable`.
_FINAL_UV_SYNC_RE = re.compile(
    r"RUN(?:\s*--mount=\S+\s*\\\n)*\s*uv sync\b.*?--no-editable", re.DOTALL
)
_STAGE_RE = re.compile(r"^FROM\s+.*\bAS\s+(\S+)", re.MULTILINE)


def _dockerfile_text() -> str:
    return (_REPO_ROOT / "Dockerfile").read_text()


def _dockerfile_stages(text: str) -> dict[str, str]:
    """Split the Dockerfile into per-stage text, keyed by stage name.

    A regex scanning the whole file cannot tell which stage a match landed
    in, so a Dockerfile that sets `RESOLVEKIT_CACHE_DIR` twice in the build
    stage and never in the runtime stage would still produce two matches
    and pass a check that only counts occurrences. Splitting on `FROM ...
    AS <stage>` boundaries first keeps every subsequent check scoped to the
    stage it claims to test.
    """
    matches = list(_STAGE_RE.finditer(text))
    assert matches, "no `FROM ... AS <stage>` line found"
    stages = {}
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        stages[match.group(1)] = text[start:end]
    return stages


def test_resolvekit_cache_dir_is_consistent_and_warmed_after_the_final_uv_sync():
    stages = _dockerfile_stages(_dockerfile_text())
    assert "build" in stages and "runtime" in stages, (
        f"expected a `build` and a `runtime` stage, found {sorted(stages)}"
    )

    cache_dir_by_stage = {}
    for stage_name in ("build", "runtime"):
        values = _ENV_CACHE_DIR_RE.findall(stages[stage_name])
        assert len(values) == 1, (
            f"expected RESOLVEKIT_CACHE_DIR set exactly once in the {stage_name} "
            f"stage, found {values}"
        )
        cache_dir_by_stage[stage_name] = values[0]
    assert cache_dir_by_stage["build"] == cache_dir_by_stage["runtime"], (
        "build and runtime stages disagree on RESOLVEKIT_CACHE_DIR "
        f"({cache_dir_by_stage['build']!r} vs {cache_dir_by_stage['runtime']!r}); "
        "COPY --from=build would land the cache at a path the runtime ENV "
        "never points at"
    )

    build_text = stages["build"]
    uv_sync_match = _FINAL_UV_SYNC_RE.search(build_text)
    warm_match = _WARM_RUN_RE.search(build_text)
    assert uv_sync_match is not None, (
        "no final `uv sync ... --no-editable` found in the build stage"
    )
    assert warm_match is not None, (
        "no RUN of scripts/warm_resolvekit_cache.py found in the build stage"
    )
    assert warm_match.start() > uv_sync_match.start(), (
        "the resolvekit warm RUN must come after the final uv sync, or it "
        "warms against a project install the build hasn't finished yet"
    )


def test_warm_script_refuses_without_resolvekit_cache_dir():
    env = dict(os.environ)
    env.pop("RESOLVEKIT_CACHE_DIR", None)
    result = subprocess.run(
        [sys.executable, str(_WARM_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode != 0
    assert "RESOLVEKIT_CACHE_DIR" in result.stderr


def test_header_journal_mode_reads_the_byte_the_runtime_depends_on(tmp_path):
    warm = _warm_module()
    wal_db = tmp_path / "wal.sqlite"
    rollback_db = tmp_path / "rollback.sqlite"
    for db_path, mode in ((wal_db, "WAL"), (rollback_db, "DELETE")):
        conn = sqlite3.connect(db_path)
        conn.execute(f"PRAGMA journal_mode={mode}")
        conn.execute("CREATE TABLE t(x)")
        conn.commit()
        conn.close()

    assert warm._header_journal_mode(wal_db) != warm._ROLLBACK_JOURNAL_HEADER_BYTE
    assert warm._header_journal_mode(rollback_db) == warm._ROLLBACK_JOURNAL_HEADER_BYTE


def test_stray_artifact_sweep_catches_every_sidecar_that_must_not_be_baked(tmp_path):
    warm = _warm_module()
    (tmp_path / "composed").mkdir()
    (tmp_path / "composed" / "store.sqlite").write_bytes(b"")
    for name in ("store.sqlite-wal", "store.sqlite-shm", "store.sqlite-journal", "store.tmp.1a2b"):
        (tmp_path / "composed" / name).write_bytes(b"")

    found = {p.name for p in warm._stray_sqlite_artifacts(tmp_path)}

    # `-journal` is the sidecar of the mode the script converts *to*, so it is
    # the one a reader would otherwise treat as hot and try to roll back.
    assert found == {
        "store.sqlite-wal",
        "store.sqlite-shm",
        "store.sqlite-journal",
        "store.tmp.1a2b",
    }


def test_stray_artifact_sweep_passes_on_a_clean_cache(tmp_path):
    warm = _warm_module()
    (tmp_path / "composed").mkdir()
    (tmp_path / "composed" / "store.sqlite").write_bytes(b"")

    assert warm._stray_sqlite_artifacts(tmp_path) == []


def test_warm_script_composes_over_entity_resolutions_own_module_list():
    """The bake script must compose the same module set the runtime
    resolver does -- `entity_resolution.shared.MODULE_IDS` -- not a
    hand-duplicated list, so the two can never drift apart.
    """
    warm = _warm_module()
    source = inspect.getsource(warm.main)
    assert "from entity_resolution.shared import MODULE_IDS" in source
    assert "list(MODULE_IDS)" in source


def test_convert_byod_pack_leaves_entities_sqlite_rollback_journal_with_no_sidecars(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Runs `_convert_byod_pack` over a real `dfo_graph` pack built by
    `entity_resolution.pack.build_pack` into `tmp_path`, and asserts the
    conversion actually flips the on-disk journal-mode header and leaves no
    WAL sidecar behind -- the property the runtime image depends on to open
    this file on Cloud Run's read-only filesystem.
    """
    monkeypatch.setenv("RESOLVEKIT_OFFLINE", "1")
    from entity_resolution import pack

    warm = _warm_module()
    pack_dir = pack.build_pack(
        records_path=pack.records_path(), curated_path=pack.curated_path(), out_dir=tmp_path
    )
    db_path = pack_dir / "entities.sqlite"
    assert warm._header_journal_mode(db_path) != warm._ROLLBACK_JOURNAL_HEADER_BYTE, (
        "a freshly built BYOD pack is expected to start in WAL mode"
    )

    warm._convert_byod_pack(tmp_path)

    assert warm._header_journal_mode(db_path) == warm._ROLLBACK_JOURNAL_HEADER_BYTE
    assert warm._stray_sqlite_artifacts(pack_dir) == []
