"""Prepare resolver SQLite packs for Cloud Run's read-only filesystem.

WAL packs need writable sidecars. This build step converts them to DELETE
journal mode, so runtime reads use the main SQLite file alone.
Domains with multiple modules use a composed cache. Single-module domains
open `entities.sqlite` directly. Convert files after resolver connections
close. Bundled packs validate SHA-256 from `metadata.json` on load, so update
the checksum after conversion. The `dfo_graph` BYOD metadata sets `checksums` to `null`.

OCI images retain mtimes at whole-second precision, and the composed-store
cache key includes source mtimes. Floor bundled pack mtimes before composing.
Floor converted single-pack files again because conversion updates mtime.

Build the same module set as `acquire_shared()` with `warm=False`. The
resolver's default starts an unjoined daemon thread that could outlive the
Docker build. Geo alone warms the compiled fuzzy-index cache. This module set loads other domains.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

# SQLite header byte 18 records journal mode: 1 for rollback, 2 for WAL.
# Read it directly because opening a connection can alter or recover the file.
_HEADER_OFFSET_JOURNAL_MODE = 18
_ROLLBACK_JOURNAL_HEADER_BYTE = 1

_COMPOSED_SUBDIR = "composed"


def _cache_dir() -> Path:
    # The Dockerfile copies this cache. Resolvekit's user cache stays in the build stage.
    raw = os.environ.get("RESOLVEKIT_CACHE_DIR", "").strip()
    if not raw:
        raise SystemExit(
            "RESOLVEKIT_CACHE_DIR must be set -- refusing to warm resolvekit's "
            "default cache location."
        )
    return Path(raw)


def _convert_to_rollback_journal(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.commit()
    finally:
        conn.close()


def _header_journal_mode(db_path: Path) -> int:
    with db_path.open("rb") as fh:
        fh.seek(_HEADER_OFFSET_JOURNAL_MODE)
        return fh.read(1)[0]


def _assert_rollback_journal(db_path: Path) -> None:
    header_mode = _header_journal_mode(db_path)
    if header_mode != _ROLLBACK_JOURNAL_HEADER_BYTE:
        raise SystemExit(
            f"{db_path} header byte {_HEADER_OFFSET_JOURNAL_MODE} is "
            f"{header_mode!r} after conversion, expected "
            f"{_ROLLBACK_JOURNAL_HEADER_BYTE!r} (rollback journal)"
        )


def _stray_sqlite_artifacts(directory: Path) -> list[Path]:
    """Return SQLite sidecars and temp files to exclude from the image.

    SQLite rolls back a hot `-journal` on open, which requires write access.
    """
    patterns = ("*-wal", "*-shm", "*-journal", "*.tmp*")
    return sorted(p for pattern in patterns for p in directory.rglob(pattern))


def _assert_no_stray_artifacts(directory: Path) -> None:
    stray = _stray_sqlite_artifacts(directory)
    if stray:
        raise SystemExit(f"stray WAL sidecar or temp file left in {directory}: {stray}")


def _assert_world_readable(db_path: Path) -> None:
    mode = db_path.stat().st_mode
    if not mode & 0o004:
        raise SystemExit(f"{db_path} is not world-readable (mode {oct(mode)})")


def _assert_world_traversable(directory: Path) -> None:
    # Cloud Run reads as uid 1000, so each parent directory needs the "other" execute bit.
    dir_mode = directory.stat().st_mode
    if not dir_mode & 0o001:
        raise SystemExit(
            f"{directory} is not world-executable (mode {oct(dir_mode)}), "
            "so uid 1000 cannot traverse into it at runtime"
        )


def _refuse_if_hardlinked(db_file: Path) -> None:
    stat = db_file.stat()
    if stat.st_nlink > 1:
        # uv's default hardlink install mode (this Docker build opts out
        # via UV_LINK_MODE=copy) can point this same inode at the
        # identical package installed for every other project sharing
        # uv's package cache. Converting or flooring the mtime of this file
        # in place would mutate theirs too, silently, the moment this script
        # runs by hand against a developer's own venv.
        raise SystemExit(
            f"{db_file} has {stat.st_nlink} hard links -- refusing to modify it "
            "in place, which would mutate the shared inode and corrupt every "
            "other project's copy in uv's package cache"
        )


def _floor_mtime(db_file: Path) -> None:
    _refuse_if_hardlinked(db_file)
    stat = db_file.stat()
    os.utime(db_file, (stat.st_atime, int(stat.st_mtime)))


def _floor_pack_mtimes(package_dir: Path) -> None:
    """Floor bundled mtimes before composition so build and runtime cache keys match.

    Conversion updates single-pack mtimes, so those files are floored again afterward.
    """
    for db_file in package_dir.rglob("*.sqlite"):
        _floor_mtime(db_file)


def _single_pack_domain_modules(module_ids: list[str]) -> list[str]:
    """Return bundled modules in single-module domains, based on resolvekit's manifest.

    The BYOD `dfo_graph` pack is absent from that manifest and is converted separately.
    """
    from resolvekit.core.module_registry import iter_manifest_entries

    domain_by_module: dict[str, str] = {}
    for module_id, _path, entry in iter_manifest_entries():
        if module_id not in module_ids:
            continue
        domain = entry.get("domain")
        if not isinstance(domain, str):
            raise SystemExit(f"manifest entry for module {module_id!r} has no 'domain'")
        domain_by_module[module_id] = domain

    domain_counts: dict[str, int] = {}
    for domain in domain_by_module.values():
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    return [
        module_id for module_id, domain in domain_by_module.items() if domain_counts[domain] == 1
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_sqlite_checksum(module_dir: Path, db_path: Path) -> None:
    """Update `metadata.json` with the converted file's checksum.

    `Resolver.lite()` verifies bundled SQLite checksums on every load.
    """
    meta_path = module_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    checksums = metadata.get("checksums")
    if not isinstance(checksums, dict) or "sqlite" not in checksums:
        raise SystemExit(f"{meta_path} has no checksums.sqlite entry to update")
    checksums["sqlite"] = _sha256(db_path)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _convert_single_pack_modules(module_ids: list[str]) -> None:
    """Convert single-pack SQLite files after `Resolver.lite()` closes its connections."""
    from resolvekit.core.module_registry import get_module_path

    for module_id in _single_pack_domain_modules(module_ids):
        module_dir = get_module_path(module_id)
        db_files = sorted(module_dir.glob("*.sqlite"))
        if len(db_files) != 1:
            raise SystemExit(
                f"expected exactly one *.sqlite file under {module_dir} for "
                f"module {module_id!r}, found {len(db_files)}: {db_files}"
            )
        db_path = db_files[0]

        if _header_journal_mode(db_path) == _ROLLBACK_JOURNAL_HEADER_BYTE:
            continue

        _refuse_if_hardlinked(db_path)
        _convert_to_rollback_journal(db_path)
        _assert_rollback_journal(db_path)
        _assert_no_stray_artifacts(module_dir)
        _assert_world_readable(db_path)
        _assert_world_traversable(module_dir)
        _rewrite_sqlite_checksum(module_dir, db_path)
        _floor_mtime(db_path)


def _convert_byod_pack(cache_dir: Path) -> None:
    """Convert the BYOD pack's SQLite file. Its cached builder returns the pack directory."""
    from entity_resolution import pack

    pack_dir = pack.build_pack(
        records_path=pack.records_path(), curated_path=pack.curated_path(), out_dir=cache_dir
    )
    db_path = pack_dir / "entities.sqlite"

    # The BYOD builder creates mode 0700. Runtime uid 1000 needs execute permission.
    # Bundled module directories retain the wheel's 0755 permissions.
    pack_dir.chmod(pack_dir.stat().st_mode | 0o755)

    if _header_journal_mode(db_path) == _ROLLBACK_JOURNAL_HEADER_BYTE:
        return

    _refuse_if_hardlinked(db_path)
    _convert_to_rollback_journal(db_path)
    _assert_rollback_journal(db_path)
    _assert_no_stray_artifacts(pack_dir)
    _assert_world_readable(db_path)
    _assert_world_traversable(pack_dir)
    _assert_world_traversable(pack_dir.parent)  # <cache_dir>/byod
    _floor_mtime(db_path)


def _convert_composed_store(cache_dir: Path) -> None:
    composed_dir = cache_dir / _COMPOSED_SUBDIR
    composed_files = sorted(composed_dir.glob("*.sqlite"))
    if len(composed_files) != 1:
        raise SystemExit(
            f"expected exactly one composed store under {composed_dir}, "
            f"found {len(composed_files)}: {composed_files}"
        )
    db_path = composed_files[0]

    _convert_to_rollback_journal(db_path)
    _assert_rollback_journal(db_path)
    _assert_no_stray_artifacts(cache_dir)
    _assert_world_readable(db_path)
    _assert_world_traversable(cache_dir)
    _assert_world_traversable(composed_dir)

    print(f"warmed and converted {db_path} to rollback-journal mode", file=sys.stderr)


def main() -> None:
    cache_dir = _cache_dir()

    # All resolver packs are local, so a manual build needs no network.
    os.environ.setdefault("RESOLVEKIT_OFFLINE", "1")

    import resolvekit
    from resolvekit.core.api import RoutingMode
    from resolvekit.core.module_registry import register_module

    from entity_resolution import pack
    from entity_resolution.shared import MODULE_IDS

    module_ids = list(MODULE_IDS)

    _floor_pack_mtimes(Path(resolvekit.__file__).parent)

    pack_dir = pack.build_pack(
        records_path=pack.records_path(), curated_path=pack.curated_path(), out_dir=cache_dir
    )
    register_module(pack.MODULE_ID, pack_dir)
    resolver = resolvekit.Resolver.lite(
        module_ids=module_ids,
        routing_mode=RoutingMode.EXPLICIT,
        cache_size=0,
        warm=False,
    )
    resolver.close()

    _convert_single_pack_modules(module_ids)
    _convert_byod_pack(cache_dir)
    _convert_composed_store(cache_dir)


if __name__ == "__main__":
    main()
